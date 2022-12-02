#include "PassDetails.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;
using namespace polygeist;

using FuncToCallsMap =
    llvm::SmallDenseMap<func::FuncOp, SmallVector<func::CallOp>>;
using CallToFuncMap = llvm::SmallDenseMap<func::CallOp, func::FuncOp>;

namespace {
// Simplify polygeist.subindex to memref.subview.
class SubIndexToSubViewPattern final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp op,
                                PatternRewriter &rewriter) const override {
    auto srcMemRefType = op.getSource().getType().cast<MemRefType>();
    auto resMemRefType = op.getResult().getType().cast<MemRefType>();

    // For now, restrict subview lowering to statically defined memref.
    if (!srcMemRefType.hasStaticShape() | !resMemRefType.hasStaticShape())
      return failure();
    auto srcRank = srcMemRefType.getRank();
    auto resRank = resMemRefType.getRank();

    // Build offset, sizes and strides.
    SmallVector<OpFoldResult> offsets(srcRank, rewriter.getIndexAttr(0));
    offsets[0] = op.getIndex();
    SmallVector<OpFoldResult> strides(srcRank, rewriter.getIndexAttr(1));

    SmallVector<OpFoldResult> sizes(srcRank);
    for (auto dim : llvm::enumerate(srcMemRefType.getShape())) {
      if (dim.index() == 0)
        sizes[0] = rewriter.getIndexAttr(
            srcRank > resRank ? 1 : resMemRefType.getDimSize(0));
      else
        sizes[dim.index()] = rewriter.getIndexAttr(dim.value());
    }

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, op.getSource(), offsets,
                                                   sizes, strides);
    return success();
  }
};
} // namespace

namespace {
class FuncLegalizePattern final : public OpRewritePattern<func::FuncOp> {
public:
  FuncLegalizePattern(MLIRContext *context, const FuncToCallsMap &map)
      : OpRewritePattern(context), map(map) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type, 16> inputTypes;
    for (auto call : map.lookup(op)) {
      if (inputTypes.empty())
        inputTypes.append(call.operand_type_begin(), call.operand_type_end());
      else if (inputTypes != call.getCalleeType().getInputs())
        return failure();
    }

    if (!inputTypes.empty())
      for (auto t : llvm::zip(op.getArguments(), inputTypes))
        std::get<0>(t).setType(std::get<1>(t));
    else
      inputTypes.append(op.getArgumentTypes().begin(),
                        op.getArgumentTypes().end());

    op.setType(rewriter.getFunctionType(
        inputTypes, op.back().getTerminator()->getOperandTypes()));
    return success();
  }

private:
  const FuncToCallsMap &map;
};
} // namespace

namespace {
class CallLegalizePattern final : public OpRewritePattern<func::CallOp> {
public:
  CallLegalizePattern(MLIRContext *context, const CallToFuncMap &map)
      : OpRewritePattern(context), map(map) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto func = map.lookup(op);
    assert(func && "function definition not found");
    for (auto t : llvm::zip(op.getResults(), func.getResultTypes()))
      std::get<0>(t).setType(std::get<1>(t));
    return success();
  }

private:
  const CallToFuncMap &map;
};
} // namespace

namespace {
struct SubIndexToSubView : public SubIndexToSubViewBase<SubIndexToSubView> {
  void runOnOperation() override {
    auto module = getOperation();
    auto context = module->getContext();

    module.walk([&](func::CallOp call) {
      auto func = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      funcToCallsMap[func].push_back(call);
      callToFuncMap[call] = func;
    });

    mlir::RewritePatternSet patterns(context);
    patterns.add<SubIndexToSubViewPattern>(context);
    patterns.add<FuncLegalizePattern>(context, funcToCallsMap);
    patterns.add<CallLegalizePattern>(context, callToFuncMap);
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }

private:
  FuncToCallsMap funcToCallsMap;
  CallToFuncMap callToFuncMap;
};
} // namespace

std::unique_ptr<Pass> mlir::polygeist::createSubIndexToSubViewPass() {
  return std::make_unique<SubIndexToSubView>();
}
