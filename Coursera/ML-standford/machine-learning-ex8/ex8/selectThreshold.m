function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
precision = 0;
recall = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    ypred = (pval < epsilon);
    tp = sum((ypred == 1) & (yval == 1));
    fp = sum((ypred == 1) & (yval == 0));
    fn = sum((ypred == 0) & (yval == 1));
    
    if (tp+fp) > 0
      precision = tp / (tp + fp);
    else
      precision = 0;
    end
    if (tp + fn) > 0
      recall = tp / (tp + fn);
     else 
      recall = 0;
     end
    
    if (precision + recall) > 0 
      F1 = 2 * (precision * recall) / (precision + recall);
    else
      F1 = 0
    end
    
    disp("=====================");
    disp(tp);
    disp(fp);
    disp(fn);
    disp(F1);
    disp("=====================");
    
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
